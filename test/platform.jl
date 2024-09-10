@testset "Platform" begin
    @testset "Info" begin
        @test length(cl.platforms()) == cl.num_platforms()

        @test cl.platform() != nothing
        @test pointer(cl.platform()) != C_NULL
        v = opencl_version(cl.platform())
        @test 1 <= v.major <= 3
        @test 0 <= v.minor <= 2
    end

    @testset "Equality" begin
        platform       = cl.platforms()[1]
        platform_copy  = cl.platforms()[1]

        @test pointer(platform) == pointer(platform_copy)
        @test hash(platform) == hash(platform_copy)
        @test isequal(platform, platform)

        if length(cl.platforms()) > 1
            p1 = cl.platforms()[1]
            for p2 in cl.platforms()[2:end]
                @test pointer(p2) != pointer(p1)
                @test hash(p2) != hash(p1)
                @test !isequal(p2, p1)
            end
        end
    end
end